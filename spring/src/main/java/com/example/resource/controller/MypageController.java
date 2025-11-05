package com.example.resource.controller;

import com.example.resource.dto.ResultDTO;
import com.example.resource.entity.Member;
import com.example.resource.security.MemberDetails;
import com.example.resource.service.MyPageService;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;

@Controller
public class MypageController {

    private final MyPageService myPageService;

    public MypageController(MyPageService myPageService) {
        this.myPageService = myPageService;
    }

    @GetMapping("/mypage")
    public String mypage(Model model, @AuthenticationPrincipal MemberDetails memberDetails) {
            Member member = memberDetails.getMember();

            List<ResultDTO> resultsList = myPageService.getResultList(member);

            model.addAttribute("memberDTO", member);
            model.addAttribute("resultsList", resultsList);
            model.addAttribute("isMypage", true);

            return "mypage";

    }
}