#version 450

// Input from the vertex shader (will be interpolated)
layout(location = 0) in vec3 fragColor;

// Output color for the current pixel
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}