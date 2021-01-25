#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#include <cstdlib>
#include <cstdio>

#include <cuda_gl_interop.h>

#include "volrend/cuda/common.cuh"
#include "volrend/renderer.hpp"
#include "volrend/n3tree.hpp"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

namespace volrend {

// ORIGINAL CODE FROM
// https://gist.github.com/allanmac/4ff11985c3562830989f

// FPS COUNTER FROM HERE:
// http://antongerdelan.net/opengl/glcontext2.html

namespace {

#define GET_RENDERER(window) \
    (*((CUDAVolumeRenderer*)glfwGetWindowUserPointer(window)))

void glfw_update_title(GLFWwindow* window) {
    // static fps counters
    static double stamp_prev = 0.0;
    static int frame_count = 0;

    // locals
    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;

    if (elapsed > 0.5) {
        stamp_prev = stamp_curr;

        const double fps = (double)frame_count / elapsed;

        char tmp[128];
        sprintf(tmp, "volrend viewer - FPS: %.2f", fps);
        glfwSetWindowTitle(window, tmp);
        // glfwSetWindowTitle(window, "volrend viewer");
        frame_count = 0;
    }

    frame_count++;
}

void draw_imgui(CUDAVolumeRenderer& rend) {
    auto& cam = rend.camera;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(20.f, 20.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(250.f, 150.f), ImGuiCond_Once);
    ImGui::Begin("Camera");

    // Update vectors indirectly since we need to normalize on change
    // (press update button) and it would be too confusing to keep normalizing
    static glm::vec3 world_down_tmp = rend.camera.v_world_down;
    static glm::vec3 world_down_prev = rend.camera.v_world_down;
    static glm::vec3 forward_tmp = rend.camera.v_forward;
    static glm::vec3 forward_prev = rend.camera.v_forward;
    if (cam.v_world_down != world_down_prev)
        world_down_tmp = world_down_prev = cam.v_world_down;
    if (cam.v_forward != forward_prev)
        forward_tmp = forward_prev = cam.v_forward;

    ImGui::InputFloat3("center", glm::value_ptr(cam.center));
    ImGui::SliderFloat("focal", &cam.focal, 100.f, 2000.f);
    ImGui::Spacing();
    ImGui::InputFloat3("world_down", glm::value_ptr(world_down_tmp));
    ImGui::InputFloat3("forward", glm::value_ptr(forward_tmp));
    if (ImGui::Button("update dirs")) {
        cam.v_world_down = glm::normalize(world_down_tmp);
        cam.v_forward = glm::normalize(forward_tmp);
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Key 1-6: preset cams");
    ImGui::End();

    // Render window
    ImGui::SetNextWindowPos(ImVec2(20.f, 180.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(250.f, 150.f), ImGuiCond_Once);
    ImGui::Begin("Rendering");

    static float inv_step_size = 1.0f / rend.options.step_size;
    if (ImGui::SliderFloat("1/step_size", &inv_step_size, 128.f, 5000.f)) {
        rend.options.step_size = 1.f / inv_step_size;
    }
    ImGui::SliderFloat("sigma_thresh", &rend.options.sigma_thresh, 0.f, 50.0f);
    ImGui::SliderFloat("stop_thresh", &rend.options.stop_thresh, 0.001f, 0.2f);
    ImGui::SliderFloat("bg_brightness", &rend.options.background_brightness,
                       0.f, 1.0f);

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void glfw_error_callback(int error, const char* description) {
    fputs(description, stderr);
}

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action,
                       int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        auto& rend = GET_RENDERER(window);
        auto& cam = rend.camera;
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_W:
            case GLFW_KEY_S:
            case GLFW_KEY_A:
            case GLFW_KEY_D:
            case GLFW_KEY_E:
            case GLFW_KEY_Q: {
                float speed = 0.005f;
                if (mods & GLFW_MOD_SHIFT) speed *= 2.f;
                if (key == GLFW_KEY_S || key == GLFW_KEY_A || key == GLFW_KEY_E)
                    speed = -speed;
                const auto& vec = (key == GLFW_KEY_A || key == GLFW_KEY_D)
                                      ? cam.v_right
                                      : (key == GLFW_KEY_W || key == GLFW_KEY_S)
                                            ? cam.v_forward
                                            : cam.v_down;
                cam.center += vec * speed;
            } break;

            case GLFW_KEY_MINUS:
                cam.focal *= 0.99f;
                break;

            case GLFW_KEY_EQUAL:
                cam.focal *= 1.01f;
                break;

            case GLFW_KEY_0:
                cam.focal = CAMERA_DEFAULT_FOCAL_LENGTH;
                break;

            case GLFW_KEY_1:
                cam.center = {0.5f, 0.0f, 0.5f};
                cam.v_forward = {0.0f, 1.0f, 0.0f};
                cam.v_world_down = {0.0f, 0.0f, -1.0f};
                break;

            case GLFW_KEY_2:
                cam.center = {0.5f, 1.0f, 0.5f};
                cam.v_forward = {0.0f, -1.0f, 0.0f};
                cam.v_world_down = {0.0f, 0.0f, 1.0f};
                break;

            case GLFW_KEY_3:
                cam.center = {0.0f, 0.5f, 0.5f};
                cam.v_forward = {1.0f, 0.0f, 0.0f};
                cam.v_world_down = {0.0f, 1.0f, 0.0f};
                break;

            case GLFW_KEY_4:
                cam.center = {1.0f, 0.5f, 0.5f};
                cam.v_forward = {-1.0f, 0.0f, 0.0f};
                cam.v_world_down = {0.0f, -1.0f, 0.0f};
                break;

            case GLFW_KEY_5:
                cam.center = {0.5f, 0.5f, 1.0f};
                cam.v_forward = {0.0f, 0.0f, -1.0f};
                cam.v_world_down = {1.0f, 0.0f, 0.0f};
                break;

            case GLFW_KEY_6:
                cam.center = {0.5f, 0.5f, 0.0f};
                cam.v_forward = {0.0f, 0.0f, 1.0f};
                cam.v_world_down = {-1.0f, 0.0f, 0.0f};
                break;
        }
    }
}

void glfw_mouse_button_callback(GLFWwindow* window, int button, int action,
                                int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto& rend = GET_RENDERER(window);
    auto& cam = rend.camera;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (action == GLFW_PRESS) {
        cam.begin_drag(
            x, y, (mods & GLFW_MOD_SHIFT) || button == GLFW_MOUSE_BUTTON_MIDDLE,
            button == GLFW_MOUSE_BUTTON_RIGHT);
    } else if (action == GLFW_RELEASE) {
        cam.end_drag();
    }
}

void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y) {
    GET_RENDERER(window).camera.drag_update(x, y);
}

void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto& cam = GET_RENDERER(window).camera;
    cam.focal *= (yoffset > 0.f) ? 1.01f : 0.99f;
}

GLFWwindow* glfw_init(const int width, const int height) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window =
        glfwCreateWindow(width, height, "GLFW / CUDA Interop", NULL, NULL);

    if (window == nullptr) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    // set up GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fputs("GLEW init failed\n", stderr);
        getchar();
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    // ignore vsync for now
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    char* glsl_version = NULL;
    ImGui_ImplOpenGL3_Init(glsl_version);
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    return window;
}

void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
    GET_RENDERER(window).resize(width, height);
}

}  // namespace
}  // namespace volrend

int main(int argc, char* argv[]) {
    using namespace volrend;
    if (argc <= 1) {
        fprintf(stderr, "Expect argument: npz file\n");
        return 1;
    }
    GLFWwindow* window = glfw_init(960, 1039);
    GLint gl_device_id;
    GLuint gl_device_count;
    cuda(GLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll));

    int cuda_device_id = (argc > 1) ? atoi(argv[1]) : gl_device_id;
    cuda(SetDevice(cuda_device_id));

    // MULTI-GPU?
    const bool multi_gpu = gl_device_id != cuda_device_id;

    // INFO
    struct cudaDeviceProp props;

    cuda(GetDeviceProperties(&props, gl_device_id));
    printf("OpenGL : %-24s (%d)\n", props.name, props.multiProcessorCount);

    cuda(GetDeviceProperties(&props, cuda_device_id));
    printf("CUDA   : %-24s (%d)\n", props.name, props.multiProcessorCount);

    {
        N3Tree tree(argv[1]);
        CUDAVolumeRenderer rend;

        // get initial width/height
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            rend.resize(width, height);
        }

        // SET USER POINTER AND CALLBACKS
        glfwSetWindowUserPointer(window, &rend);
        glfwSetKeyCallback(window, glfw_key_callback);
        glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
        glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);
        glfwSetScrollCallback(window, glfw_scroll_callback);
        glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

        // LOOP UNTIL DONE
        while (!glfwWindowShouldClose(window)) {
            // MONITOR FPS
            glfw_update_title(window);

            rend.clear();

            rend.render(tree);
            rend.swap();

            draw_imgui(rend);

            glfwSwapBuffers(window);
            glFinish();
            // glfwPollEvents();
            glfwWaitEvents();
            // break;
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    cuda(DeviceReset());
}
