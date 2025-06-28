// example.frag
#version 450
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 outColor;


void main() {
    outColor = fragColor;
}