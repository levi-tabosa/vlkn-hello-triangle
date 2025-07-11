// test.vert
#version 450

// Set 0: Is shared with text3d.vert 
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_offset;
layout(location = 2) in vec4 in_color;
layout(location = 3) out vec4 out_color;
void main() {
    gl_Position =  ubo.projection * ubo.view * vec4(in_pos + in_offset, 1.0);
    out_color = in_color;
}