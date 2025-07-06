// text3d.vert
#version 450

// The same UBO you're likely using for other 3D objects
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;
} ubo;

// --- Vertex Attributes ---
// Local position of the character quad's vertex (e.g., top-left corner is at (0,0,0))
layout(location = 0) in vec3 in_local_pos; 
// UV coordinate in the font atlas
layout(location = 1) in vec2 in_uv;
// Color for the text
layout(location = 2) in vec4 in_color;
// The model matrix for this entire string, passed per-vertex
layout(location = 3) in mat4 in_model;

// --- Outputs to Fragment Shader ---
layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;

void main() {
    // Transform the local vertex position by the string's model matrix, then by view and projection
    gl_Position = ubo.projection * ubo.view * in_model * vec4(in_local_pos, 1.0);
    
    frag_uv = in_uv;
    frag_color = in_color;
}