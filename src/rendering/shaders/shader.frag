R"(

#version 460 core

in vec2 uv;
out vec4 fragment_output;
uniform sampler2D framebuffer;

void main() {
    fragment_output = texture2D(framebuffer, uv);
}

)"
