package com.example.resource.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class TestController {

    @GetMapping("test")
    public String test_g(){
        return "test_graph";
    }

    @GetMapping("test/morphing")
    public String test_m(){
        return "test_morphing";
    }
}
