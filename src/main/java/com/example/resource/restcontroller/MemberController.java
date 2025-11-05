package com.example.resource.restcontroller;

import com.example.resource.dto.AccountActionResponse;
import com.example.resource.security.MemberDetails;
import com.example.resource.service.MemberService;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;

@RestController
@RequestMapping("/member")
public class MemberController {

    private final MemberService memberService;

    public MemberController(MemberService memberService) {
        this.memberService = memberService;
    }

    @PostMapping("/check/{id}")
    public AccountActionResponse checkMember(@PathVariable Long id,
                                             @AuthenticationPrincipal MemberDetails memberDetails,
                                             @RequestParam("password") String inputPassword) {
        memberService.checkPassword(id, memberDetails.getId(), inputPassword);
        return new AccountActionResponse(true, "");
    }

    @PostMapping("/change/pw/{id}")
    public AccountActionResponse updatePW(@PathVariable Long id,
                                          @AuthenticationPrincipal MemberDetails memberDetails,
                                          @RequestParam("password") String newPassword) {
        memberService.updatePW(id, memberDetails.getId(), newPassword);
        return new AccountActionResponse(true, "비밀번호가 변경되었습니다.");
    }

    @PostMapping("/change/bd/{id}")
    public AccountActionResponse changeBD(@PathVariable Long id,
                                          @AuthenticationPrincipal MemberDetails memberDetails,
                                          @RequestParam LocalDate birthday) {
        memberService.updateBirthDay(id, memberDetails.getId(), birthday);
        return new AccountActionResponse(true, "생년월일이 수정되었습니다.");
    }

    @PostMapping("/cancel/{id}")
    public AccountActionResponse cancelAccount(@PathVariable Long id,
                                               @AuthenticationPrincipal MemberDetails memberDetails) {
        memberService.cancelAccount(id, memberDetails.getId());
        return new AccountActionResponse(true, "회원 탈퇴가 완료되었습니다.");
    }
}