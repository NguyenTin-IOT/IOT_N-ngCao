#include <Arduino.h>
#include <driver/i2s.h>
#include <Lab3_Audio_ESP32_inferencing.h>

#define I2S_WS   4
#define I2S_SCK  5
#define I2S_SD   6
#define I2S_PORT I2S_NUM_0

// Bộ đệm chứa đủ 1 khung hình âm thanh suy luận (ví dụ 16000 mẫu = 1 giây ở 16kHz)
static int16_t audio_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
static volatile bool is_inferencing = false;
static volatile bool record_ready = false;

// Hàm callback nạp dữ liệu âm thanh chuẩn hóa [-1.0, 1.0] cho mô hình Edge Impulse
int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)audio_buffer[offset + i] / 32768.0f;
    }
    return 0;
}

// Task suy luận AI (Chạy độc lập trên Core 1)
void tflite_inference_task(void *pvParameters) {
  while (true) {
    if (record_ready && !is_inferencing) {
      is_inferencing = true;
      
      signal_t features_signal;
      features_signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
      features_signal.get_data = &microphone_audio_signal_get_data;      
      
      ei_impulse_result_t result = { 0 };
      EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
      
      if (res == EI_IMPULSE_OK) {
        Serial.println("\n------------------------------------");
        Serial.printf("Thời gian DSP: %d ms | Phân loại: %d ms\n", result.timing.dsp, result.timing.classification);
        
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            Serial.printf("%-12s: %.2f%%\n", result.classification[ix].label, result.classification[ix].value * 100.0f);
        }
        
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        if (result.anomaly >= 0.3) {
            Serial.printf("=> CẢNH BÁO BẤT THƯỜNG! (Score: %.3f)\n", result.anomaly);
        }
#endif
        Serial.println("------------------------------------");
      }
      
      record_ready = false;
      is_inferencing = false;
    }
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

void setup() {
  Serial.begin(115200);
  
  unsigned long start = millis();
  while (!Serial && (millis() - start < 3000)) {
    delay(10);
  }

  Serial.println("\n=== KHỞI ĐỘNG HỆ THỐNG TINYML PHÂN LOẠI ÂM THANH ===");
  Serial.printf("Kích thước khung hình mẫu: %d samples\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);

  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = 16000,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 512,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0
  };
  
  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = -1,
      .data_in_num = I2S_SD
  };
  
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_start(I2S_PORT);

  // Tạo Task AI chạy trên Core 1
  xTaskCreatePinnedToCore(tflite_inference_task, "Inference_Task", 16384, NULL, 1, NULL, 1);
}

void loop() {
  if (!record_ready && !is_inferencing) {
    size_t bytesRead = 0;
    // Đọc đủ số mẫu âm thanh I2S cho 1 khung hình suy luận
    esp_err_t result = i2s_read(I2S_PORT, (void*)audio_buffer, sizeof(audio_buffer), &bytesRead, portMAX_DELAY);
    if (result == ESP_OK && bytesRead == sizeof(audio_buffer)) {
      record_ready = true;
    }
  }
  vTaskDelay(1 / portTICK_PERIOD_MS);
}