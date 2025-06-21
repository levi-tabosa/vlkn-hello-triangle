#version 450

// The UBO, with the same structure as the vertex shader
layout(binding = 0) uniform UniformBufferObject {
    vec4 color;
    vec2 offset; // Unused in this shader, but must be declared to match the layout
} ubo;

// The output color that will be written to the framebuffer
layout(location = 0) out vec4 out_color;

void main() {
    out_color = ubo.color;
}
