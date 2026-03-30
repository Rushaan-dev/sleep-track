#define SAMPLE_RATE 512
#define INPUT_PIN A0

unsigned long lastSampleTime = 0;
const unsigned long sampleInterval = 1000000 / SAMPLE_RATE;

void setup() {
  Serial.begin(115200);
}

void loop() {
  unsigned long currentTime = micros();
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    int eegValue = analogRead(INPUT_PIN);
    Serial.println(eegValue);
  }
}
