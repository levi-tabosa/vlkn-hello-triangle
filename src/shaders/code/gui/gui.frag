// gui.frag
#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

// New: We bind the font texture atlas at binding 0
layout(binding = 0) uniform sampler2D fontSampler;

layout(location = 0) out vec4 outColor;

void main() {
    // Sample the texture. For a single-channel font atlas, the glyph shape is in the red channel.
    float alpha = texture(fontSampler, fragUV).r;

    // Modulate the vertex color with the texture's alpha.
    // This tints the text and makes the background transparent.
    outColor = vec4(fragColor.rgb, fragColor.a * alpha + vec4(0.1) );
}