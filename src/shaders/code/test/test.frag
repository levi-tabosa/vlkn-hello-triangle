// test.frag
// This is a test fragment shader.
// used in wireframe rendering
# version 450

layout(location = 1) in vec4 in_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = in_color;
}
