#include <stdio.h>
#include <stdlib.h>
#include "esp_camera.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_system.h"
#include "esp_event.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "driver/gpio.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "esp_http_server.h"
#include "tool/dl_tool.hpp"
#include "model_define.hpp"
#include <string.h>

#define ESP32_S3_CAM

#ifdef ESP32_S3_CAM
#define CAM_PIN_PWDN -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK 15
#define CAM_PIN_SIOD 4
#define CAM_PIN_SIOC 5
#define CAM_PIN_D7 16
#define CAM_PIN_D6 17
#define CAM_PIN_D5 18
#define CAM_PIN_D4 12
#define CAM_PIN_D3 10
#define CAM_PIN_D2 8
#define CAM_PIN_D1 9
#define CAM_PIN_D0 11
#define CAM_PIN_VSYNC 6
#define CAM_PIN_HREF 7
#define CAM_PIN_PCLK 13
#endif

const int input_height = 96;
const int input_width = 96;
const int input_channel = 1;
const int input_exponent = -13;

const char* ssid = "xxxx";
const char* password = "xxxx";

static const char* TAG = "app";
httpd_handle_t server = NULL;
static EventGroupHandle_t wifi_event_group;
const int WIFI_CONNECTED_BIT = BIT0;

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected from Wi-Fi. Reconnecting...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP Address: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

esp_err_t start_wifi() {
    esp_err_t ret;

    wifi_event_group = xEventGroupCreate();

    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_config_t wifi_config;
    memset(&wifi_config, 0, sizeof(wifi_config));
    strncpy((char *)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, password, sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    EventBits_t bits = xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, pdMS_TO_TICKS(10000));
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Wi-Fi connected successfully.");
    } else {
        ESP_LOGE(TAG, "Failed to connect to Wi-Fi within 10 seconds.");
        return ESP_FAIL;
    }

    return ESP_OK;
}

void init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_QQVGA;  
    config.jpeg_quality = 20;             
    config.fb_count = 1;                  

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
    }
}

void preprocess_image(camera_fb_t* fb, int16_t* output) {
    int width = fb->width;
    int height = fb->height;
    int stride = width;

    for (int y = 0; y < input_height; y++) {
        for (int x = 0; x < input_width; x++) {
            int orig_x = x * width / input_width;
            int orig_y = y * height / input_height;
            output[y * input_width + x] = fb->buf[orig_y * stride + orig_x];
        }
    }
}

esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Failed to capture image");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // 进行模型推理
    int16_t example_element[96 * 96] = {0};
    preprocess_image(fb, example_element);

    HANDRECOGNITION model;
    Tensor<int16_t> input;
    input.set_element((int16_t *)example_element)
         .set_exponent(input_exponent)
         .set_shape({input_height, input_width, input_channel})
         .set_auto_free(false);
    model.forward(input);

    float *score = model.l14.get_output().get_element_ptr();
    float max_score = score[0];
    int max_index = 0;

    for (size_t i = 0; i < 10; i++) {
        if (score[i] > max_score) {
            max_score = score[i];
            max_index = i;
        }
    }

    const char* result;
    switch (max_index) {
        case 0: result = "01_palm"; break;
        case 1: result = "02_I"; break;
        case 2: result = "03_fist"; break;
        case 3: result = "04_fist_moved"; break;
        case 4: result = "05_thumb"; break;
        case 5: result = "06_index"; break;
        case 6: result = "07_ok"; break;
        case 7: result = "08_palm_moved"; break;
        case 8: result = "09_c"; break;
        case 9: result = "10_down"; break;
        default: result = "No result";
    }

    char response[200];
    snprintf(response, sizeof(response), "<html><body><h2>Hand Gesture: %s</h2><img src=\"data:image/jpeg;base64,", result);

    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, response, strlen(response));
    
    esp_camera_fb_return(fb);
    return ESP_OK;
}

esp_err_t root_handler(httpd_req_t *req) {
    const char* response = "<html><body>"
                            "<h1>ESP32 Camera</h1>"
                            "<img src=\"/capture\" id=\"capture\"/><br/><br/>"
                            "<button onclick=\"refreshImage()\">Capture and Analyze</button>"
                            "<div id=\"result\"></div>"
                            "<script>"
                            "function refreshImage() {"
                            "  var img = document.getElementById('capture');"
                            "  img.src = '/capture?' + new Date().getTime();"
                            "}"
                            "</script>"
                            "</body></html>";
    httpd_resp_send(req, response, strlen(response));
    return ESP_OK;
}

void start_webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    ESP_ERROR_CHECK(httpd_start(&server, &config));

    httpd_uri_t uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = root_handler,
        .user_ctx = NULL
    };
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &uri));

    uri.uri = "/capture";
    uri.handler = capture_handler;
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &uri));
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "Starting...");

    ESP_ERROR_CHECK(start_wifi());
    init_camera();
    start_webserver();

    while (true) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}
