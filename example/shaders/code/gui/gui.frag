// gui.frag
#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(binding = 0) uniform sampler2D fontSampler;

layout(location = 0) out vec4 outColor;

void main() {
    float alpha = texture(fontSampler, fragUV).r;
    // Modulate the vertex color with the texture's alpha.
    // This tints the text and makes the background transparent.
    outColor = vec4(fragColor.rgb, fragColor.a * alpha );

}