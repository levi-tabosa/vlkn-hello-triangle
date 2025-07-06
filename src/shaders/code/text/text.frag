// text3d.frag
#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

// This is now correct. It samples from the descriptor bound to SET 1, BINDING 0.
layout(set = 1, binding = 0) uniform sampler2D fontSampler;

layout(location = 0) out vec4 outColor;

void main() {
    float alpha = texture(fontSampler, fragUV).r;
    outColor = vec4(fragColor.rgb, fragColor.a * alpha);
}