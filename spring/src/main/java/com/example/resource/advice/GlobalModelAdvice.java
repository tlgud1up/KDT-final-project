package com.example.resource.advice;

import com.example.resource.dto.MemberDTO;
import com.example.resource.security.MemberDetails;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.web.csrf.CsrfToken;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ModelAttribute;

@ControllerAdvice
public class GlobalModelAdvice {

    @ModelAttribute("_csrf")
    public CsrfToken csrfToken(HttpServletRequest request) {
        return (CsrfToken) request.getAttribute("_csrf");
    }

    @ModelAttribute("memberDTO")
    public MemberDTO addMemberToModel(@AuthenticationPrincipal MemberDetails memberDetails) {
        if (memberDetails == null) return null;

        return MemberDTO.builder()
                .id(memberDetails.getId())
                .name(memberDetails.getName())
                .username(memberDetails.getUsername())
                .birthday(memberDetails.getBirthday())
                .joinDate(memberDetails.getJoinDate())
                .userType(memberDetails.getUserType())
                .build();
    }
}