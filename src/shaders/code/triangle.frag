#version 450

// Output color for the fragment
layout(location = 0) out vec4 outColor;

// Uniform Buffer Object, containing our color data
layout(binding = 0) uniform UniformBufferObject {
    vec4 color;
} ubo;

void main() {
    // Set the output color to the color from the UBO
    outColor = ubo.color;
}