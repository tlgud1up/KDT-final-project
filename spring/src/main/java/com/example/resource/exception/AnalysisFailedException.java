package com.example.resource.exception;

public class AnalysisFailedException extends RuntimeException {
    public AnalysisFailedException(String message) {
        super(message);
    }
}