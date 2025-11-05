package com.example.resource.controller;

import com.example.resource.dto.ResultDTO;
import com.example.resource.entity.AnalysisResult;
import com.example.resource.entity.Member;
import com.example.resource.entity.OrigImage;
import com.example.resource.repository.AnalysisResultRepository;
import com.example.resource.repository.OrigImageRepository;
import com.example.resource.service.MapperService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.security.Principal;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@RestController
@RequiredArgsConstructor
public class ResultRestController {

    private final OrigImageRepository origImageRepository;
    private final AnalysisResultRepository analysisResultRepository;
    private final MapperService mapperService;

    @GetMapping("/results")
    public List<ResultDTO> getAllResults(Principal principal) {
        return origImageRepository.findAll()
                .stream()
                .filter(img -> img.getMember().getUsername().equals(principal.getName()))
                .sorted((a, b) -> b.getAnalysisDate().compareTo(a.getAnalysisDate()))
                .map(img -> {
                    AnalysisResult result = analysisResultRepository.findByOrigImgId(img.getId());
                    return result != null ? mapperService.toResultDTO(result, img) : null;
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }
}