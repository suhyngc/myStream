import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper as h
from onnx import numpy_helper as nph
from onnx import shape_inference

def create_onnx_transformer(
    model_path="transformer_model.onnx",
    num_blocks=6,
    batch_size=1,
    sequence_length=128,
    embedding_size=768,
    num_heads=12,
    ff_hidden_size=3072,
):
    """
    Creates and saves a transformer-like ONNX model with a specified number of blocks.

    Each block consists of a multi-head self-attention layer followed by a feed-forward network.
    Residual connections and layer normalization are applied after each sub-layer.

    Args:
        model_path (str): The path to save the generated .onnx file.
        num_blocks (int): The number of times to repeat the attention and feed-forward block.
        batch_size (int): The batch size for the input tensor.
        sequence_length (int): The sequence length for the input tensor.
        embedding_size (int): The dimension of the input and output embeddings.
        num_heads (int): The number of heads for multi-head self-attention.
        ff_hidden_size (int): The hidden size of the feed-forward network.
    """
    if embedding_size % num_heads != 0:
        raise ValueError("embedding_size must be divisible by num_heads.")

    head_size = embedding_size // num_heads

    # --- Graph Inputs ---
    graph_input = h.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_size, sequence_length, embedding_size]
    )

    nodes = []
    initializers = []
    current_input = "input"

    for i in range(num_blocks):
        block_prefix = f"block_{i}"

        # --- 1. Multi-Head Self-Attention ---
        # Layer Norm before Attention
        ln1_scale = nph.from_array(np.ones(embedding_size, dtype=np.float32), name=f"{block_prefix}_ln1_scale")
        ln1_bias = nph.from_array(np.zeros(embedding_size, dtype=np.float32), name=f"{block_prefix}_ln1_bias")
        initializers.extend([ln1_scale, ln1_bias])

        ln1_output = f"{block_prefix}_ln1_output"
        nodes.append(h.make_node(
            "LayerNormalization",
            inputs=[current_input, ln1_scale.name, ln1_bias.name],
            outputs=[ln1_output],
            name=f"{block_prefix}_ln1",
            epsilon=1e-5
        ))

        # QKV linear projection
        qkv_weights = nph.from_array(
            np.random.randn(embedding_size, embedding_size * 3).astype(np.float32),
            name=f"{block_prefix}_qkv_weights"
        )
        qkv_bias = nph.from_array(
            np.zeros(embedding_size * 3, dtype=np.float32), name=f"{block_prefix}_qkv_bias"
        )
        initializers.extend([qkv_weights, qkv_bias])

        qkv_matmul_output = f"{block_prefix}_qkv_matmul_output"
        nodes.append(h.make_node(
            "MatMul",
            inputs=[ln1_output, qkv_weights.name],
            outputs=[qkv_matmul_output],
            name=f"{block_prefix}_qkv_matmul"
        ))
        
        qkv_add_output = f"{block_prefix}_qkv_add_output"
        nodes.append(h.make_node(
            "Add",
            inputs=[qkv_matmul_output, qkv_bias.name],
            outputs=[qkv_add_output],
            name=f"{block_prefix}_qkv_add"
        ))

        # Split Q, K, V
        q_out, k_out, v_out = [f"{block_prefix}_{n}" for n in ["q", "k", "v"]]
        # Add explicit split sizes for compatibility with tools expecting split input
        split_sizes = nph.from_array(np.array([embedding_size, embedding_size, embedding_size], dtype=np.int64), name=f"{block_prefix}_qkv_split_sizes")
        initializers.append(split_sizes)
        nodes.append(h.make_node(
            "Split",
            inputs=[qkv_add_output, split_sizes.name],
            outputs=[q_out, k_out, v_out],
            name=f"{block_prefix}_qkv_split",
            axis=2
        ))
        
        # Reshape for multi-head
        # New shape: [batch_size, sequence_length, num_heads, head_size]
        reshape_shape = nph.from_array(np.array([batch_size, sequence_length, num_heads, head_size], dtype=np.int64), name=f"{block_prefix}_reshape_shape")
        initializers.append(reshape_shape)
        
        q_reshaped, k_reshaped, v_reshaped = [f"{n}_reshaped" for n in [q_out, k_out, v_out]]
        nodes.extend([
            h.make_node("Reshape", [q_out, reshape_shape.name], [q_reshaped], name=f"{q_out}_reshape"),
            h.make_node("Reshape", [k_out, reshape_shape.name], [k_reshaped], name=f"{k_out}_reshape"),
            h.make_node("Reshape", [v_out, reshape_shape.name], [v_reshaped], name=f"{v_out}_reshape")
        ])

        # Transpose for batch matmul
        # New shape: [batch_size, num_heads, sequence_length, head_size]
        q_transposed, k_transposed, v_transposed = [f"{n}_transposed" for n in [q_reshaped, k_reshaped, v_reshaped]]
        nodes.extend([
            h.make_node("Transpose", [q_reshaped], [q_transposed], perm=[0, 2, 1, 3], name=f"{q_reshaped}_transpose"),
            h.make_node("Transpose", [k_reshaped], [k_transposed], perm=[0, 2, 1, 3], name=f"{k_reshaped}_transpose"),
            h.make_node("Transpose", [v_reshaped], [v_transposed], perm=[0, 2, 1, 3], name=f"{v_reshaped}_transpose")
        ])

        # Scaled dot-product attention
        scaling_factor = nph.from_array(np.array(1.0 / np.sqrt(head_size), dtype=np.float32), name=f"{block_prefix}_scaling_factor")
        initializers.append(scaling_factor)

        k_transposed_for_matmul = f"{k_transposed}_for_matmul"
        nodes.append(h.make_node("Transpose", [k_transposed], [k_transposed_for_matmul], perm=[0, 1, 3, 2], name=f"{k_transposed}_transpose_matmul"))

        qk_matmul = f"{block_prefix}_qk_matmul"
        nodes.append(h.make_node("MatMul", [q_transposed, k_transposed_for_matmul], [qk_matmul], name=f"{block_prefix}_qk_matmul_node"))
        
        qk_scaled = f"{block_prefix}_qk_scaled"
        nodes.append(h.make_node("Mul", [qk_matmul, scaling_factor.name], [qk_scaled], name=f"{block_prefix}_qk_scale_node"))
        
        qk_softmax = f"{block_prefix}_qk_softmax"
        nodes.append(h.make_node("Softmax", [qk_scaled], [qk_softmax], axis=-1, name=f"{block_prefix}_qk_softmax_node"))

        qkv_attention = f"{block_prefix}_qkv_attention"
        nodes.append(h.make_node("MatMul", [qk_softmax, v_transposed], [qkv_attention], name=f"{block_prefix}_qkv_attention_node"))
        
        # Transpose and reshape back
        attention_transposed = f"{block_prefix}_attention_transposed"
        nodes.append(h.make_node("Transpose", [qkv_attention], [attention_transposed], perm=[0, 2, 1, 3], name=f"{block_prefix}_attention_transpose_back"))

        attention_reshaped = f"{block_prefix}_attention_reshaped"
        final_reshape_shape = nph.from_array(np.array([batch_size, sequence_length, embedding_size], dtype=np.int64), name=f"{block_prefix}_final_reshape_shape")
        initializers.append(final_reshape_shape)
        nodes.append(h.make_node("Reshape", [attention_transposed, final_reshape_shape.name], [attention_reshaped], name=f"{block_prefix}_attention_reshape_back"))
        
        # Final projection
        proj_weights = nph.from_array(np.random.randn(embedding_size, embedding_size).astype(np.float32), name=f"{block_prefix}_proj_weights")
        proj_bias = nph.from_array(np.zeros(embedding_size, dtype=np.float32), name=f"{block_prefix}_proj_bias")
        initializers.extend([proj_weights, proj_bias])

        proj_matmul = f"{block_prefix}_proj_matmul"
        nodes.append(h.make_node("MatMul", [attention_reshaped, proj_weights.name], [proj_matmul], name=f"{block_prefix}_proj_matmul_node"))
        
        attention_output = f"{block_prefix}_attention_output"
        nodes.append(h.make_node("Add", [proj_matmul, proj_bias.name], [attention_output], name=f"{block_prefix}_proj_add_node"))

        # Residual connection
        residual1_output = f"{block_prefix}_residual1"
        nodes.append(h.make_node("Add", [current_input, attention_output], [residual1_output], name=f"{block_prefix}_residual1_add"))

        # --- 2. Feed-Forward Network ---
        # Layer Norm before FFN
        ln2_scale = nph.from_array(np.ones(embedding_size, dtype=np.float32), name=f"{block_prefix}_ln2_scale")
        ln2_bias = nph.from_array(np.zeros(embedding_size, dtype=np.float32), name=f"{block_prefix}_ln2_bias")
        initializers.extend([ln2_scale, ln2_bias])
        
        ln2_output = f"{block_prefix}_ln2_output"
        nodes.append(h.make_node(
            "LayerNormalization",
            inputs=[residual1_output, ln2_scale.name, ln2_bias.name],
            outputs=[ln2_output],
            name=f"{block_prefix}_ln2",
            epsilon=1e-5
        ))

        # FFN Layer 1
        ffn1_weights = nph.from_array(np.random.randn(embedding_size, ff_hidden_size).astype(np.float32), name=f"{block_prefix}_ffn1_weights")
        ffn1_bias = nph.from_array(np.zeros(ff_hidden_size, dtype=np.float32), name=f"{block_prefix}_ffn1_bias")
        initializers.extend([ffn1_weights, ffn1_bias])
        
        ffn1_matmul = f"{block_prefix}_ffn1_matmul"
        nodes.append(h.make_node("MatMul", [ln2_output, ffn1_weights.name], [ffn1_matmul], name=f"{block_prefix}_ffn1_matmul_node"))
        
        ffn1_add = f"{block_prefix}_ffn1_add"
        nodes.append(h.make_node("Add", [ffn1_matmul, ffn1_bias.name], [ffn1_add], name=f"{block_prefix}_ffn1_add_node"))
        
        ffn1_relu = f"{block_prefix}_ffn1_relu"
        nodes.append(h.make_node("Relu", [ffn1_add], [ffn1_relu], name=f"{block_prefix}_ffn1_relu_node"))

        # FFN Layer 2
        ffn2_weights = nph.from_array(np.random.randn(ff_hidden_size, embedding_size).astype(np.float32), name=f"{block_prefix}_ffn2_weights")
        ffn2_bias = nph.from_array(np.zeros(embedding_size, dtype=np.float32), name=f"{block_prefix}_ffn2_bias")
        initializers.extend([ffn2_weights, ffn2_bias])

        ffn2_matmul = f"{block_prefix}_ffn2_matmul"
        nodes.append(h.make_node("MatMul", [ffn1_relu, ffn2_weights.name], [ffn2_matmul], name=f"{block_prefix}_ffn2_matmul_node"))
        
        ffn_output = f"{block_prefix}_ffn_output"
        nodes.append(h.make_node("Add", [ffn2_matmul, ffn2_bias.name], [ffn_output], name=f"{block_prefix}_ffn2_add_node"))

        # Residual connection 2
        residual2_output = f"{block_prefix}_residual2"
        nodes.append(h.make_node("Add", [residual1_output, ffn_output], [residual2_output], name=f"{block_prefix}_residual2_add"))

        # Update current_input for the next block
        current_input = residual2_output


    # --- Graph Outputs ---
    graph_output = h.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, sequence_length, embedding_size]
    )
    
    # Final LayerNorm
    final_ln_scale = nph.from_array(np.ones(embedding_size, dtype=np.float32), name="final_ln_scale")
    final_ln_bias = nph.from_array(np.zeros(embedding_size, dtype=np.float32), name="final_ln_bias")
    initializers.extend([final_ln_scale, final_ln_bias])

    nodes.append(h.make_node(
        "LayerNormalization",
        inputs=[current_input, final_ln_scale.name, final_ln_bias.name],
        outputs=["output"],
        name="final_ln",
        epsilon=1e-5
    ))

    # --- Construct the Graph and Model ---
    graph_def = h.make_graph(
        nodes,
        "transformer-graph",
        [graph_input],
        [graph_output],
        initializer=initializers,
    )

    model_def = h.make_model(graph_def, producer_name="onnx-transformer-generator")
    model_def.opset_import[0].version = 17

    # --- Apply Shape Inference ---
    model_def = shape_inference.infer_shapes(model_def)
    
    # Copy inferred value_info back to the graph to ensure all intermediate tensors have type info
    # This is needed for compatibility with some tools
    graph_def.ClearField('value_info')
    graph_def.value_info.extend(model_def.graph.value_info)

    # --- Validate and Save ---
    onnx.checker.check_model(model_def)
    onnx.save(model_def, model_path)
    print(f"Model saved to {model_path}")


def create_onnx_transformer_simplified(
    model_path="transformer_model_simplified.onnx",
    num_blocks=6,
    batch_size=1,
    sequence_length=128,
    embedding_size=768,
    num_heads=12,
    ff_hidden_size=3072,
):
    """
    Creates and saves a simplified transformer-like ONNX model without non-linear operations.
    
    Removes: Add (bias and residual), Softmax, ReLU, and LayerNormalization.
    Keeps only: MatMul, Transpose, Reshape, Split, and Mul operations.
    
    This is useful for traffic estimation and dataflow analysis without actual computation.

    Args:
        model_path (str): The path to save the generated .onnx file.
        num_blocks (int): The number of times to repeat the attention and feed-forward block.
        batch_size (int): The batch size for the input tensor.
        sequence_length (int): The sequence length for the input tensor.
        embedding_size (int): The dimension of the input and output embeddings.
        num_heads (int): The number of heads for multi-head self-attention.
        ff_hidden_size (int): The hidden size of the feed-forward network.
    """
    if embedding_size % num_heads != 0:
        raise ValueError("embedding_size must be divisible by num_heads.")

    head_size = embedding_size // num_heads

    # --- Graph Inputs ---
    graph_input = h.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_size, sequence_length, embedding_size]
    )

    nodes = []
    initializers = []
    current_input = "input"

    for i in range(num_blocks):
        block_prefix = f"block_{i}"

        # --- 1. Multi-Head Self-Attention (Simplified) ---
        # Skip LayerNormalization
        
        # QKV linear projection (MatMul only, no bias Add)
        qkv_weights = nph.from_array(
            np.random.randn(embedding_size, embedding_size * 3).astype(np.float32),
            name=f"{block_prefix}_qkv_weights"
        )
        initializers.append(qkv_weights)

        qkv_matmul_output = f"{block_prefix}_qkv_matmul_output"
        nodes.append(h.make_node(
            "MatMul",
            inputs=[current_input, qkv_weights.name],
            outputs=[qkv_matmul_output],
            name=f"{block_prefix}_qkv_matmul"
        ))
        
        # Skip bias Add

        # Split Q, K, V
        q_out, k_out, v_out = [f"{block_prefix}_{n}" for n in ["q", "k", "v"]]
        # Add explicit split sizes for compatibility with tools expecting split input
        split_sizes = nph.from_array(np.array([embedding_size, embedding_size, embedding_size], dtype=np.int64), name=f"{block_prefix}_qkv_split_sizes")
        initializers.append(split_sizes)
        nodes.append(h.make_node(
            "Split",
            inputs=[qkv_matmul_output, split_sizes.name],
            outputs=[q_out, k_out, v_out],
            name=f"{block_prefix}_qkv_split",
            axis=2
        ))
        
        # Reshape for multi-head
        # New shape: [batch_size, sequence_length, num_heads, head_size]
        reshape_shape = nph.from_array(np.array([batch_size, sequence_length, num_heads, head_size], dtype=np.int64), name=f"{block_prefix}_reshape_shape")
        initializers.append(reshape_shape)
        
        q_reshaped, k_reshaped, v_reshaped = [f"{n}_reshaped" for n in [q_out, k_out, v_out]]
        nodes.extend([
            h.make_node("Reshape", [q_out, reshape_shape.name], [q_reshaped], name=f"{q_out}_reshape"),
            h.make_node("Reshape", [k_out, reshape_shape.name], [k_reshaped], name=f"{k_out}_reshape"),
            h.make_node("Reshape", [v_out, reshape_shape.name], [v_reshaped], name=f"{v_out}_reshape")
        ])

        # Transpose for batch matmul
        # New shape: [batch_size, num_heads, sequence_length, head_size]
        q_transposed, k_transposed, v_transposed = [f"{n}_transposed" for n in [q_reshaped, k_reshaped, v_reshaped]]
        nodes.extend([
            h.make_node("Transpose", [q_reshaped], [q_transposed], perm=[0, 2, 1, 3], name=f"{q_reshaped}_transpose"),
            h.make_node("Transpose", [k_reshaped], [k_transposed], perm=[0, 2, 1, 3], name=f"{k_reshaped}_transpose"),
            h.make_node("Transpose", [v_reshaped], [v_transposed], perm=[0, 2, 1, 3], name=f"{v_reshaped}_transpose")
        ])

        # Scaled dot-product attention (without Softmax)
        scaling_factor = nph.from_array(np.array(1.0 / np.sqrt(head_size), dtype=np.float32), name=f"{block_prefix}_scaling_factor")
        initializers.append(scaling_factor)

        k_transposed_for_matmul = f"{k_transposed}_for_matmul"
        nodes.append(h.make_node("Transpose", [k_transposed], [k_transposed_for_matmul], perm=[0, 1, 3, 2], name=f"{k_transposed}_transpose_matmul"))

        qk_matmul = f"{block_prefix}_qk_matmul"
        nodes.append(h.make_node("MatMul", [q_transposed, k_transposed_for_matmul], [qk_matmul], name=f"{block_prefix}_qk_matmul_node"))
        
        qk_scaled = f"{block_prefix}_qk_scaled"
        nodes.append(h.make_node("Mul", [qk_matmul, scaling_factor.name], [qk_scaled], name=f"{block_prefix}_qk_scale_node"))
        
        # Skip Softmax

        qkv_attention = f"{block_prefix}_qkv_attention"
        nodes.append(h.make_node("MatMul", [qk_scaled, v_transposed], [qkv_attention], name=f"{block_prefix}_qkv_attention_node"))
        
        # Transpose and reshape back
        attention_transposed = f"{block_prefix}_attention_transposed"
        nodes.append(h.make_node("Transpose", [qkv_attention], [attention_transposed], perm=[0, 2, 1, 3], name=f"{block_prefix}_attention_transpose_back"))

        attention_reshaped = f"{block_prefix}_attention_reshaped"
        final_reshape_shape = nph.from_array(np.array([batch_size, sequence_length, embedding_size], dtype=np.int64), name=f"{block_prefix}_final_reshape_shape")
        initializers.append(final_reshape_shape)
        nodes.append(h.make_node("Reshape", [attention_transposed, final_reshape_shape.name], [attention_reshaped], name=f"{block_prefix}_attention_reshape_back"))
        
        # Final projection (MatMul only, no bias Add)
        proj_weights = nph.from_array(np.random.randn(embedding_size, embedding_size).astype(np.float32), name=f"{block_prefix}_proj_weights")
        initializers.append(proj_weights)

        attention_output = f"{block_prefix}_attention_output"
        nodes.append(h.make_node("MatMul", [attention_reshaped, proj_weights.name], [attention_output], name=f"{block_prefix}_proj_matmul_node"))
        
        # Skip bias Add
        # Skip residual connection

        # --- 2. Feed-Forward Network (Simplified) ---
        # Skip LayerNormalization

        # FFN Layer 1 (MatMul only, no bias Add, no ReLU)
        ffn1_weights = nph.from_array(np.random.randn(embedding_size, ff_hidden_size).astype(np.float32), name=f"{block_prefix}_ffn1_weights")
        initializers.append(ffn1_weights)
        
        ffn1_matmul = f"{block_prefix}_ffn1_matmul"
        nodes.append(h.make_node("MatMul", [attention_output, ffn1_weights.name], [ffn1_matmul], name=f"{block_prefix}_ffn1_matmul_node"))
        
        # Skip bias Add
        # Skip ReLU

        # FFN Layer 2 (MatMul only, no bias Add)
        ffn2_weights = nph.from_array(np.random.randn(ff_hidden_size, embedding_size).astype(np.float32), name=f"{block_prefix}_ffn2_weights")
        initializers.append(ffn2_weights)

        # For the last block, output directly to "output", otherwise use intermediate name
        if i == num_blocks - 1:
            ffn_output = "output"
        else:
            ffn_output = f"{block_prefix}_ffn_output"
            
        nodes.append(h.make_node("MatMul", [ffn1_matmul, ffn2_weights.name], [ffn_output], name=f"{block_prefix}_ffn2_matmul_node"))
        
        # Skip bias Add
        # Skip residual connection

        # Update current_input for the next block
        current_input = ffn_output


    # --- Graph Outputs ---
    graph_output = h.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, sequence_length, embedding_size]
    )
    
    # --- Construct the Graph and Model ---
    graph_def = h.make_graph(
        nodes,
        "transformer-graph-simplified",
        [graph_input],
        [graph_output],
        initializer=initializers,
    )

    model_def = h.make_model(graph_def, producer_name="onnx-transformer-generator-simplified")
    model_def.opset_import[0].version = 17

    # --- Apply Shape Inference ---
    model_def = shape_inference.infer_shapes(model_def)
    
    # Copy inferred value_info back to the graph to ensure all intermediate tensors have type info
    # This is needed for compatibility with some tools
    graph_def.ClearField('value_info')
    graph_def.value_info.extend(model_def.graph.value_info)

    # --- Validate and Save ---
    onnx.checker.check_model(model_def)
    onnx.save(model_def, model_path)
    print(f"Simplified model saved to {model_path}")


if __name__ == "__main__":
    # You can customize your model architecture here
    
    # Create full transformer model
    create_onnx_transformer(
        model_path="my_custom_model.onnx",
        num_blocks=1,         # How many times to repeat the self-attention + FFN block
        sequence_length=128,  # The length of the input sequence
        embedding_size=256,   # The main dimension of the model
        num_heads=8,          # Number of attention heads (must divide embedding_size)
        ff_hidden_size=2048,  # Hidden layer size in the feed-forward network
    )
    
    # Create simplified transformer model (for traffic estimation)
    create_onnx_transformer_simplified(
        model_path="my_custom_model_simplified.onnx",
        num_blocks=1,         # How many times to repeat the self-attention + FFN block
        sequence_length=128,  # The length of the input sequence
        embedding_size=256,   # The main dimension of the model
        num_heads=8,          # Number of attention heads (must divide embedding_size)
        ff_hidden_size=2048,  # Hidden layer size in the feed-forward network
    )
