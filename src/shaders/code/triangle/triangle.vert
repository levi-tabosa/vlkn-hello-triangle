#version 450
layout(location = 0) in vec2 in_pos;

layout(binding = 0) uniform UniformBufferObject {
    vec4 color;  // Unused
    vec2 offset; 
} ubo;

void main() {

    gl_Position = vec4(in_pos + ubo.offset, 0.0, 1.0);
}
