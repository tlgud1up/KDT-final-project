package com.example.resource.security;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;
import org.springframework.security.web.savedrequest.HttpSessionRequestCache;
import org.springframework.security.web.savedrequest.SavedRequest;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
public class LoginFailureHandler implements AuthenticationFailureHandler {

    @Override
    public void onAuthenticationFailure(HttpServletRequest request,
                                        HttpServletResponse response,
                                        AuthenticationException exception) throws IOException, ServletException {

        String errorMessage;

        if (exception instanceof BadCredentialsException) {
            errorMessage = "아이디와 비밀번호가 일치하지 않습니다.";
        } else {
            errorMessage = "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.";
        }

        // 저장된 요청 가져오기 (redirect 전에!)
        HttpSessionRequestCache requestCache = new HttpSessionRequestCache();
        SavedRequest savedRequest = requestCache.getRequest(request, response);

        // 저장된 요청이 있으면 세션에 별도로 보관
        if (savedRequest != null) {
            String targetUrl = savedRequest.getRedirectUrl();
            request.getSession().setAttribute("SPRING_SECURITY_SAVED_REQUEST_URL", targetUrl);
        }

        request.getSession().setAttribute("errorMessage", errorMessage);
        response.sendRedirect("/login?error");
    }
}