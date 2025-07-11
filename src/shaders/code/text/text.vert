// text3d.vet
#version 450

// Set 0: Scene UBO (same as test.vert)
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;
} ubo;

// Set 2: SSBO with an array of model matrices
layout(set = 2, binding = 0) readonly buffer DrawDataTable {
    mat4 models[];
} drawData;

layout(location = 0) in vec3 in_local_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

// --- Outputs to Fragment Shader ---
layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;

void main() {
    // Get the model matrix for this specific draw call using gl_InstanceIndex.
    // gl_InstanceIndex gets its value from the `firstInstance` field in the
    // VkDrawIndexedIndirectCommand struct.
    mat4 model = drawData.models[gl_InstanceIndex];

    // Transform the local vertex position by the looked-up model matrix
    gl_Position = ubo.projection * ubo.view * model * vec4(in_local_pos, 1.0);

    frag_uv = in_uv;
    frag_color = in_color;
}