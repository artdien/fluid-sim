R"(

#version 460 core

out vec2 uv;

const vec2[6] quad_position = {
    {-1.0, +1.0}, {+1.0, -1.0}, {+1.0, +1.0}, // Triangle 1
    {-1.0, +1.0}, {-1.0, -1.0}, {+1.0, -1.0}, // Triangle 2
};

const vec2[6] quad_uv = {
    { 0.0, +1.0}, {+1.0,  0.0}, {+1.0, +1.0}, // Triangle 1
    { 0.0, +1.0}, { 0.0,  0.0}, {+1.0,  0.0}, // Triangle 2
};

void main() {
    gl_Position = vec4(quad_position[gl_VertexID], 0.0, 1.0);
    uv = quad_uv[gl_VertexID];
}

)"

