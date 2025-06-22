// example.vert
#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;
    vec4 color;
} ubo;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_offset; // <-- Add this

void main() {
    gl_Position =  ubo.projection * ubo.view * vec4(in_pos + in_offset, 1.0);
}