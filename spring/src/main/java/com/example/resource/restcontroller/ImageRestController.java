
package com.example.resource.restcontroller;


import com.example.resource.dto.ImageValidDTO;
import com.example.resource.security.MemberDetails;
import com.example.resource.service.ImageAnalysisService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Mono;

import java.util.Map;

@Slf4j
@RestController
@RequiredArgsConstructor
public class ImageRestController {
    private final ImageAnalysisService imageAnalysisService;

    @PostMapping("/api/images/validate")
    public Mono<ResponseEntity<ImageValidDTO>> validateImage(
            @RequestParam("image") MultipartFile file,
            @AuthenticationPrincipal MemberDetails memberDetails) {

        if (memberDetails == null) {
            return Mono.just(ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(new ImageValidDTO("1", "로그인이 필요합니다")));
        }

        return imageAnalysisService.validateImage(file)
                .map(result -> {
                    if ("0".equals(result.getStatus())) {
                        return ResponseEntity.ok(result);
                    } else {
                        return ResponseEntity.badRequest().body(result);
                    }
                })
                .onErrorResume(e -> Mono.just(ResponseEntity.status(500)
                        .body(new ImageValidDTO("1", e.getMessage()))));
    }


    @PostMapping("/api/images/upload")
    public Mono<ResponseEntity<Map<String, Object>>> uploadImage(@RequestParam("image") MultipartFile file,
                                                                 @AuthenticationPrincipal MemberDetails memberDetails, Authentication authentication) {
        log.info("=== 이미지 업로드 시작 ===");
        log.info("Authentication: {}", authentication);
        log.info("memberDetails: {}", memberDetails);
        log.info("인증 여부: {}", authentication != null && authentication.isAuthenticated());

        if (memberDetails == null) {
            log.error("memberDetails가 null입니다!");
            return Mono.just(ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("status", 1, "error", "로그인이 필요합니다")));
        }
        return imageAnalysisService.analyzeImage(file, memberDetails.getMember())
                .doOnSuccess(result -> log.info("분석 성공: {}", result))
                .map(ResponseEntity::ok)
                .onErrorResume(e -> {
                    log.error("=== 이미지 분석 실패 ===");
                    log.error("에러 타입: {}", e.getClass().getName());
                    log.error("에러 메시지: {}", e.getMessage());
                    log.error("상세 스택:", e);
                    return Mono.just(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                            .body(Map.of("status", 1, "error", e.getMessage())));
                });
    }
}
