# Stage 3A Study Note: Attention From Scratch

This note explains the core ideas behind Transformer attention in beginner-friendly language.

## 1) What are Q, K, and V?
For each token embedding in a sequence, we create three new vectors using linear layers:

- **Q (Query):** what this token is looking for.
- **K (Key):** what each token offers as searchable information.
- **V (Value):** the information content to mix together.

You can think of attention as:
- Query asks: “Which tokens are relevant to me?”
- Key helps score relevance.
- Value is what gets combined after scoring.

## 2) Attention scores
The relevance between tokens is measured with a dot product:

`scores = Q @ K.transpose(-2, -1) / sqrt(d_k)`

- `Q @ K^T` compares each query token with every key token.
- Dividing by `sqrt(d_k)` keeps values numerically stable when dimensions are larger.

Resulting shape is usually:
- `(batch_size, seq_len, seq_len)` for single-head attention
- `(batch_size, num_heads, seq_len, seq_len)` for multi-head attention

## 3) Softmax weights
We convert raw scores into probabilities:

`weights = softmax(scores, dim=-1)`

- Each row sums to 1.
- Bigger weight means “pay more attention to that token.”

## 4) Context vectors
Now we blend value vectors using those weights:

`context = weights @ V`

The output `context` is a weighted summary for each token position.

## 5) Why multi-head attention?
Instead of one attention pattern, we use several heads in parallel.
Each head can learn a different relationship (for example, local context vs global context).

High-level steps:
1. Project to Q, K, V.
2. Split embeddings into heads.
3. Compute scaled dot-product attention per head.
4. Concatenate all heads.
5. Apply a final linear projection.

This gives the model multiple “views” of token relationships at the same time.

## 6) What the Stage 3A script demonstrates
The script `scripts/study_stage3_attention_from_scratch.py` prints tensor shapes for:
- input embeddings
- Q, K, V projections
- attention scores
- attention weights
- context vectors
- split heads
- concatenated heads
- final output

This is preparation for implementing the full Stage 3 Mini Transformer classifier.
