package com.example.resource.exception;

public class SignupException extends RuntimeException {
    public SignupException(String message) {
        super(message);
    }
}