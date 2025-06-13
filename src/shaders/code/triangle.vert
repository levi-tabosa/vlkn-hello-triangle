#version 450

// Input vertex data, from the vertex buffer
layout(location = 0) in vec2 inPosition;

void main() {
    // Output the position in clip space coordinates.
    // The z-coordinate is for depth, and w is for perspective division.
    gl_Position = vec4(inPosition, 0.0, 1.0);
}