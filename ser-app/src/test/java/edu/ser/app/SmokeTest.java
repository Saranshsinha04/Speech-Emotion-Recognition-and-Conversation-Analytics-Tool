package edu.ser.app;

import ai.onnxruntime.OrtEnvironment;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class SmokeTest {
    @Test
    void onnxRuntimeLoads() {
        assertNotNull(OrtEnvironment.getEnvironment().getVersion());
    }
}
