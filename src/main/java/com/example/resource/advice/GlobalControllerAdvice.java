package com.example.resource.advice;


import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ModelAttribute;

@ControllerAdvice
public class GlobalControllerAdvice     {

    @ModelAttribute("authenticated")
    public boolean addAuthentication(Authentication authentication) {
        return authentication != null && authentication.isAuthenticated();
    }
}
