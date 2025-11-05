package com.example.resource.restcontroller;

import com.example.resource.dto.MorphingRequest;
import com.example.resource.security.MemberDetails;
import com.example.resource.service.MorphingService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.Map;

@RestController
public class MorphingController {

    private final MorphingService morphingService;

    public MorphingController(MorphingService morphingService) {
        this.morphingService = morphingService;
    }

    @PostMapping("/morphing/{id}")
    public Mono<ResponseEntity<Map<String, Object>>> morphing(
            @PathVariable Long id,
            @AuthenticationPrincipal MemberDetails memberDetails,
            @RequestBody MorphingRequest request) {

        return morphingService.getMorphImageAsync(id, memberDetails.getMember(), request)
                .map(result -> Map.of(
                        "status", 0,
                        "data", result
                ))
                .map(ResponseEntity::ok)
                .onErrorResume(e -> {
                    Map<String, Object> errorResponse = Map.of(
                            "status", 1,
                            "error", e.getMessage()
                    );
                    return Mono.just(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse));
                });
    }

}