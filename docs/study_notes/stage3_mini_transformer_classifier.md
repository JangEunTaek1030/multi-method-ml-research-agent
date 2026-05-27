# Stage 3 Mini Transformer Classifier (Beginner Notes)

## How Stage 3 differs from Stage 2
Stage 2 used a simple model: token embedding -> masked mean pooling -> MLP.

Stage 3 still starts with token ids, but now each token representation is processed by Transformer encoder block(s). This means tokens can interact with each other using attention before pooling.

## Token embedding + position embedding
- **Token embedding** turns each token id into a dense vector.
- **Position embedding** adds information about token order (1st token, 2nd token, ...).
- We add token embedding and position embedding so the model knows both token identity and position.

## Why position embedding is needed
Self-attention alone does not know order. Without position embedding, the sequence "model trains fast" and "fast trains model" can look too similar. Position embeddings inject order information.

## Multi-head self-attention
For each token, self-attention builds:
- Query (Q)
- Key (K)
- Value (V)

Then attention scores are computed with `QK^T / sqrt(head_dim)`, normalized with softmax, and used to mix V vectors.

**Multi-head** means we do this in parallel across several heads, so the model can learn different relation types between tokens.

## Transformer encoder block
One encoder block has:
1. Multi-head self-attention
2. Residual connection + LayerNorm
3. Feed-forward network
4. Residual connection + LayerNorm

This keeps the architecture modular and easy to stack.

## Residual connection
Residual means adding input back to output of a sub-layer (`x + layer(x)`). This helps training stay stable and keeps useful original information.

## LayerNorm
LayerNorm normalizes features inside each token representation. It helps optimization and reduces training instability.

## Feed-forward network
After attention, each token goes through a small MLP:
- Linear up-projection
- Activation (GELU/ReLU)
- Dropout
- Linear down-projection

This lets the model perform non-linear feature transformation per token.

## Masked mean pooling for classification
For sentence-level classification, we need one vector per sequence.
- Ignore PAD tokens using a mask.
- Average only non-PAD token vectors.
- Send pooled vector to classifier head.

## Why this is still a small educational baseline
This model is intentionally lightweight:
- small embedding size
- few heads
- few layers
- tiny sample dataset

It is good for learning the mechanics of Transformers from scratch and for reproducible experiments, not for state-of-the-art performance.
