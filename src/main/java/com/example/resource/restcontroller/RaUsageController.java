package com.example.resource.restcontroller;


import com.example.resource.dto.RaUsageDTO;
import com.example.resource.service.RaUsageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;


@RestController
@RequestMapping("/api")
public class RaUsageController {

    @Autowired
    private RaUsageService raUsageService;
    @Autowired
    private WebClient.Builder webClientBuilder;

    @GetMapping("/predict")
    public List<RaUsageDTO> getPredictions() {
        return raUsageService.getDataFromDB();
    }

}
