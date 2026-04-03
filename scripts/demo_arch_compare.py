import torch
from torch import nn


class ToyVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=256, target_tokens=64):
        super().__init__()
        self.target_tokens = target_tokens
        self.merge = source_tokens // target_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * self.merge, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        batch_size, _, hidden_size = x.shape
        x = x.reshape(batch_size, self.target_tokens, hidden_size * self.merge)
        return self.mlp(x)


class ToyQFormer(nn.Module):
    def __init__(self, vision_dim=32, lm_dim=48, num_queries=8, num_heads=4):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, lm_dim))
        self.vision_proj = nn.Linear(vision_dim, lm_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=lm_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(lm_dim)

    def forward(self, vision_tokens):
        batch_size = vision_tokens.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        vision_tokens = self.vision_proj(vision_tokens)
        fused_queries, attn_weights = self.cross_attn(
            query=queries,
            key=vision_tokens,
            value=vision_tokens,
        )
        return self.norm(fused_queries), attn_weights


class ToyLanguageCrossAttention(nn.Module):
    def __init__(self, lm_dim=48, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=lm_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(lm_dim)

    def forward(self, text_states, visual_queries):
        mixed_states, attn_weights = self.cross_attn(
            query=text_states,
            key=visual_queries,
            value=visual_queries,
        )
        return self.norm(text_states + mixed_states), attn_weights


def replace_placeholder_tokens(text_states, visual_tokens, placeholder_mask):
    fused = text_states.clone()
    for batch_idx in range(text_states.size(0)):
        positions = torch.where(placeholder_mask[batch_idx])[0]
        take = min(len(positions), visual_tokens.size(1))
        fused[batch_idx, positions[:take]] = visual_tokens[batch_idx, :take]
    return fused


def print_header(title):
    print(f"\n{'=' * 16} {title} {'=' * 16}")


def main():
    torch.manual_seed(7)

    batch_size = 2
    vision_token_count = 256
    vision_dim = 32
    text_token_count = 12
    lm_dim = 48
    image_token_len = 4

    vision_tokens = torch.randn(batch_size, vision_token_count, vision_dim)
    text_states = torch.randn(batch_size, text_token_count, lm_dim)
    placeholder_mask = torch.zeros(batch_size, text_token_count, dtype=torch.bool)
    placeholder_mask[:, 3:3 + image_token_len] = True

    print_header("Input")
    print(f"vision tokens: {tuple(vision_tokens.shape)}")
    print(f"text states:   {tuple(text_states.shape)}")
    print(f"placeholder positions per sample: {placeholder_mask[0].nonzero(as_tuple=False).flatten().tolist()}")

    print_header("Route A: Q-Former + Cross-Attention")
    q_former = ToyQFormer(
        vision_dim=vision_dim,
        lm_dim=lm_dim,
        num_queries=8,
        num_heads=4,
    )
    cross_attn_block = ToyLanguageCrossAttention(lm_dim=lm_dim, num_heads=4)

    visual_queries, q_attn = q_former(vision_tokens)
    fused_text_a, txt_attn = cross_attn_block(text_states, visual_queries)

    print("step 1: learned query tokens attend to all vision tokens")
    print(f"visual queries after Q-Former: {tuple(visual_queries.shape)}")
    print(f"Q-Former attention map:        {tuple(q_attn.shape)}")
    print("step 2: text states cross-attend to visual queries inside the language stack")
    print(f"fused text states:             {tuple(fused_text_a.shape)}")
    print(f"text->vision attention map:    {tuple(txt_attn.shape)}")

    print_header("Route B: MiniMind-V Placeholder Replacement")
    projector = ToyVisionProjector(
        in_dim=vision_dim,
        out_dim=lm_dim,
        source_tokens=vision_token_count,
        target_tokens=image_token_len,
    )
    visual_slots = projector(vision_tokens)
    fused_text_b = replace_placeholder_tokens(text_states, visual_slots, placeholder_mask)

    print("step 1: project many vision tokens into a fixed number of language-sized slots")
    print(f"projected visual slots:        {tuple(visual_slots.shape)}")
    print("step 2: directly overwrite the placeholder token embeddings in the text sequence")
    print(f"fused text states:             {tuple(fused_text_b.shape)}")
    print("step 3: send the mixed sequence into a plain decoder-only transformer")
    print("extra text<->vision cross-attention block needed: False")

    print_header("Key Difference")
    print("Q-Former route keeps vision tokens separate, then injects them via cross-attention.")
    print("MiniMind-V route turns the image into a few pseudo-text embeddings and splices them into the sequence.")
    print("Q-Former route is usually more expressive but adds modules and compute.")
    print("MiniMind-V route is simpler and cheaper, but the image-text interaction is more constrained.")


if __name__ == "__main__":
    main()
