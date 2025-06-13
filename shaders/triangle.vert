#version 450

// Input vertex data, specified by location
layout(location = 0) in vec2 inPosition;
layout(location =1) in vec3 inColor;

// Output data to the fragment shader
layout(location = 0) out vec3 fragColor;

void main() {
    // gl_Position is a special required output variable
    gl_Position = vec4(inPosition, 0.0, 1.0);
    
    // Pass the input color directly to the fragment shader
    fragColor = inColor;
}