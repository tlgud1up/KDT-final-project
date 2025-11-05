package com.example.resource.security;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.SavedRequestAwareAuthenticationSuccessHandler;
import org.springframework.security.web.savedrequest.HttpSessionRequestCache;
import org.springframework.security.web.savedrequest.RequestCache;
import org.springframework.security.web.savedrequest.SavedRequest;
import org.springframework.security.web.util.matcher.AntPathRequestMatcher;
import org.springframework.security.web.util.matcher.RequestMatcher;

import java.io.IOException;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private final LoginFailureHandler loginFailureHandler;

    public SecurityConfig(LoginFailureHandler loginFailureHandler) {
        this.loginFailureHandler = loginFailureHandler;
    }


    @Bean
    SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {

        http
                .requestCache(cache -> cache
                        .requestCache(requestCache())
                )
                .formLogin((formLogin)->formLogin
                        .loginPage("/login")
                        .loginProcessingUrl("/doLogin")
                        .successHandler(savedRequestAwareAuthenticationSuccessHandler())
                        .failureHandler(loginFailureHandler)
                        .permitAll()

                )
                .logout(logout -> logout
                        .logoutRequestMatcher(new AntPathRequestMatcher("/logout", "GET")) // GET 허용
                        .logoutSuccessUrl("/")
                        .invalidateHttpSession(true)
                        .clearAuthentication(true)
                        .permitAll()
                )
                .authorizeHttpRequests(authorize -> authorize
                        .requestMatchers("/error","/css/**", "/webjars/**","/images/**","/data/**","/morphing","/favicon.ico").permitAll()
                        .requestMatchers("/", "/about").permitAll()
                        .requestMatchers("/api/**").permitAll()
                        .requestMatchers("/signup").permitAll()
                        .requestMatchers("/results").authenticated()
                        .anyRequest().authenticated());
        return http.build();
    }

        @Bean
        PasswordEncoder passwordEncoder() {
            return new BCryptPasswordEncoder();
        }

        @Bean
        public RequestCache requestCache() {
            HttpSessionRequestCache requestCache = new HttpSessionRequestCache();
            requestCache.setRequestMatcher(new RequestMatcher() {
                @Override
                public boolean matches(HttpServletRequest request) {
                    // WebSocket 요청은 저장하지 않음
                    String uri = request.getRequestURI();
                    return !uri.startsWith("/ws") &&
                           !uri.startsWith("/login") &&
                           !uri.startsWith("/favicon.ico") &&
                           !uri.startsWith("/.well-known");
                }
            });
            return requestCache;
        }

    @Bean
    public SavedRequestAwareAuthenticationSuccessHandler savedRequestAwareAuthenticationSuccessHandler() {
        SavedRequestAwareAuthenticationSuccessHandler handler = new SavedRequestAwareAuthenticationSuccessHandler();
        handler.setDefaultTargetUrl("/");
        handler.setRequestCache(requestCache());

        return new SavedRequestAwareAuthenticationSuccessHandler() {
            @Override
            public void onAuthenticationSuccess(HttpServletRequest request,
                                                HttpServletResponse response,
                                                Authentication authentication) throws IOException, ServletException {

                // 1. 세션에 백업된 URL 확인 (로그인 실패 후 성공한 경우)
                String targetUrl = (String) request.getSession()
                        .getAttribute("SPRING_SECURITY_SAVED_REQUEST_URL");

                if (targetUrl != null) {
                    request.getSession().removeAttribute("SPRING_SECURITY_SAVED_REQUEST_URL");
                    getRedirectStrategy().sendRedirect(request, response, targetUrl);
                    return;
                }

                // 2. RequestCache 확인 (로그인 실패 없이 바로 성공한 경우)
                SavedRequest savedRequest = requestCache().getRequest(request, response);
                if (savedRequest != null) {
                    String redirectUrl = savedRequest.getRedirectUrl();
                    requestCache().removeRequest(request, response);
                    getRedirectStrategy().sendRedirect(request, response, redirectUrl);
                    return;
                }

                // 3. 둘 다 없으면 홈으로
                getRedirectStrategy().sendRedirect(request, response, "/");
            }
        };
    }



}