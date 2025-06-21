#version 450

// Input vertex data, from the vertex buffer
layout(location = 0) in vec2 in_pos;

// The UBO, which will now contain our animation data
layout(binding = 0) uniform UniformBufferObject {
    vec4 color;  // Unused in this shader, but part of the shared struct
    vec2 offset; // The new offset we will apply to the vertices
} ubo;

void main() {
    // Add the offset to the input position
    // The z-component is for depth and w is for perspective division
    gl_Position = vec4(in_pos + ubo.offset, 0.0, 1.0);
}
