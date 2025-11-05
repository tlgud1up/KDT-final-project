package com.example.resource.controller;

import com.example.resource.dto.SignupRequest;
import com.example.resource.exception.SignupException;
import com.example.resource.repository.MemberRepository;
import com.example.resource.service.MemberService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
public class AuthController {

    private final MemberService memberService;

    public AuthController(MemberRepository memberRepository, MemberService memberService) {
        this.memberService = memberService;
    }

    @GetMapping("/login")
    public String loginPage(HttpServletRequest request, Model model) {
        HttpSession session = request.getSession(false);
        if (session != null) {
            String errorMsg = (String) session.getAttribute("errorMessage");
            if (errorMsg != null) {
                model.addAttribute("errorMessage", errorMsg);
                session.removeAttribute("errorMessage");
            }
        }

        return "member/login";
    }

    @GetMapping("/signup")
    public String signup(Model model) {

        SignupRequest signupRequest = SignupRequest.builder()
                .username("").password("").name("").build();

        model.addAttribute("signupRequest", signupRequest);
        return "member/signup";
    }

    @PostMapping("/signup")
    public String signup(@Valid SignupRequest signupRequest,
                         BindingResult bindingResult, Model model,
                         RedirectAttributes redirectAttributes,
                         HttpServletRequest request){
        if (bindingResult.hasErrors()) {
            model.addAttribute("errorMessage", "모든 항목을 입력해주세요.");
            model.addAttribute("signupRequest", signupRequest);
            return "member/signup";
        }

        try {
            memberService.register(signupRequest);

            redirectAttributes.addFlashAttribute("successMessage", "회원가입이 완료되었습니다.");
            return "redirect:/login";

        } catch (SignupException e) {
            model.addAttribute("errorMessage", e.getMessage());
            return "member/signup";
        }
    }
}