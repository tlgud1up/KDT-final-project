package com.example.resource.controller;


import com.example.resource.dto.ResultDTO;
import com.example.resource.entity.AnalysisResult;
import com.example.resource.entity.OrigImage;
import com.example.resource.repository.AnalysisResultRepository;
import com.example.resource.repository.OrigImageRepository;
import com.example.resource.service.MapperService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.server.ResponseStatusException;

import java.security.Principal;


@Controller
@RequiredArgsConstructor
public class ResultController {
    private final AnalysisResultRepository analysisResultRepository;
    private final OrigImageRepository origImageRepository;
    private final MapperService mapperService;

    @GetMapping("/result/{id}")
    public String getResult(@PathVariable Long id, Model model, Principal principal) {


        OrigImage origImage = origImageRepository.findById(id).orElse(null);


        if(!origImage.getMember().getUsername().equals(principal.getName())){
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "접근 권한이 없습니다.");
        }


        AnalysisResult analysisResult = analysisResultRepository.findByOrigImgId(origImage.getId());


        ResultDTO resultdto = mapperService.toResultDTO(analysisResult, origImage);


        // ✅ result.total 값 가져오기
        double total = resultdto.getTotal();  // ResultDTO에 getTotal()이 있다고 가정
        // ✅ 기준치에 따라 적합 여부 판단
        boolean isSuitable = total < 1.0;
        model.addAttribute("isSuitable", isSuitable);


        model.addAttribute("result", resultdto);


        return "result2";
    }


    @GetMapping("/result/{id}/morphing")
    public String morphing(@PathVariable Long id, Model model, Principal principal) {
        return "morphing";
    }
//    @GetMapping("/morphing")
//    public String morphing(){
//        return "morphing";
//    }

}