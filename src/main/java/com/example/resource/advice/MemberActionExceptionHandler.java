package com.example.resource.advice;

import com.example.resource.dto.AccountActionResponse;
import com.example.resource.restcontroller.MemberController;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice(assignableTypes = MemberController.class)
public class MemberActionExceptionHandler {

    @ExceptionHandler(IllegalArgumentException.class)
    public AccountActionResponse handleIllegalArgument(IllegalArgumentException e) {
        return new AccountActionResponse(false, e.getMessage());
    }

    @ExceptionHandler(Exception.class)
    public AccountActionResponse handleGeneralException(Exception e) {
        return new AccountActionResponse(false, "서버 오류가 발생했습니다. 다시 시도해주세요.");
    }
}