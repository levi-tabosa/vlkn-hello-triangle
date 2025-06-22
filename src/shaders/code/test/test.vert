// test.vert
// Vertex shader for testing purposes
#version 450

layout(location = 0) in vec3 in_pos;
layout (location = 1) in vec3 offset;
layout(location = 1) out vec4 out_color;

void main() {
    gl_Position = vec4(in_pos + offset, 1.0);
    out_color = gl_Position;
}
