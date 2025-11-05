package com.example.resource.controller;

import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomepageController {

    @GetMapping("/")
    public String homepage(Model model) {

        model.addAttribute("isHome", true);

        return "home";
    }

//    @GetMapping("/")
//    public String home() {
//        return "home";
//    }

}
