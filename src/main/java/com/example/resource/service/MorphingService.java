package com.example.resource.service;

import com.example.resource.dto.MorphingEffect;
import com.example.resource.dto.MorphingRequest;
import com.example.resource.entity.Member;
import com.example.resource.entity.OrigImage;
import com.example.resource.exception.AnalysisFailedException;
import com.example.resource.repository.OrigImageRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.io.ByteArrayInputStream;
import java.lang.reflect.Field;
import java.util.Base64;

@Slf4j
@Service
public class MorphingService {

    private final OrigImageRepository origImageRepository;
    private final MapperService mapperService;
    private final WebClient webClient;

    public MorphingService(OrigImageRepository origImageRepository,
                           WebClient imageClient, MapperService mapperService) {
        this.origImageRepository = origImageRepository;
        this.mapperService = mapperService;
        this.webClient = imageClient;
    }

    public Mono<MorphingEffect> getMorphImageAsync(Long id, Member member, MorphingRequest request) {
        return Mono.fromCallable(() -> loadAndCheck(id, member))
                .flatMap(origImage -> callPythonServer(origImage, request))
                .onErrorMap(e -> {
                    log.error("모핑 처리 실패: {}", e.getMessage(), e);
                    return new AnalysisFailedException(e.getMessage());
                });
    }

    private OrigImage loadAndCheck(Long id, Member member) {
        OrigImage origImage = origImageRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("이미지를 찾을 수 없습니다. id=" + id));

        if (!origImage.getMember().getId().equals(member.getId())) {
            throw new AccessDeniedException("접근 권한이 없습니다.");
        }
        return origImage;
    }

    private Mono<MorphingEffect> callPythonServer(OrigImage origImage, MorphingRequest request) {
        try {
            byte[] imageData = origImage.getImageData();
            if (imageData == null || imageData.length == 0) {
                throw new IllegalArgumentException("이미지 데이터가 존재하지 않습니다.");
            }

            String originDataUri = mapperService.toDataUri(imageData);

            ByteArrayInputStream bais = new ByteArrayInputStream(imageData);

            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("file", new InputStreamResource(bais))
                    .header("Content-Disposition", "form-data; name=file; filename=image.png");

            builder.part("h", request.getH());
            builder.part("s", request.getS());
            builder.part("v", request.getV());

            return webClient.post()
                    .uri("/morphing")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(builder.build()))
                    .retrieve()
                    .bodyToMono(MorphingEffect.class)
                    .map(effect -> {
                        MorphingEffect result = base64ToDataUri(effect);  // FastAPI 결과 5개 변환
                        result.setOriginImg(originDataUri);  // 원본 이미지 추가
                        return result;
                    });

        } catch (Exception e) {
            log.error("Python 모핑 호출 중 오류 발생: {}", e.getMessage());
            return Mono.error(new AnalysisFailedException(e.getMessage()));
        }
    }


    private MorphingEffect base64ToDataUri(MorphingEffect effect) {
        if (effect == null) return null;

        Field[] fields = MorphingEffect.class.getDeclaredFields();
        for (Field field : fields) {

            if (field.getName().equals("originImg")) {
                continue;
            }
            if (field.getType() == String.class) {
                field.setAccessible(true);
                try {
                    String base64 = (String) field.get(effect);
                    String dataUri = mapperService.toDataUri(Base64.getDecoder().decode(base64));
                    field.set(effect, dataUri);
                } catch (IllegalAccessException e) {
                    log.error("모핑 이미지 data uri로 변환 실패: {}", e.getMessage());
                }
            }
        }
        return effect;
    }

}