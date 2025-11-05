package com.example.resource.service;

import com.example.resource.dto.RaUsageDTO;
import com.example.resource.dto.ResultDTO;
import com.example.resource.entity.AnalysisResult;
import com.example.resource.entity.OrigImage;
import com.example.resource.entity.RaUsage;
import com.example.resource.repository.AnalysisResultRepository;
import org.springframework.stereotype.Service;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.URLConnection;
import java.time.format.DateTimeFormatter;
import java.util.Base64;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class MapperService {

    private final AnalysisResultRepository analysisResultRepository;

    public MapperService(AnalysisResultRepository analysisResultRepository) {
        this.analysisResultRepository = analysisResultRepository;
    }

    public List<RaUsageDTO> toRaUsageDTO(List<RaUsage> data) {
        return data.stream()
                .map(raUsage -> RaUsageDTO.builder()
                        .year(raUsage.getYear())
                        .sales(raUsage.getSales())
                        .build())
                .collect(Collectors.toList());
    }

    public RaUsage toRaUsage(RaUsageDTO raUsageDTO) {
        return RaUsage.builder()
                .year(raUsageDTO.getYear())
                .sales(raUsageDTO.getSales())
                .build();
    }

    public ResultDTO toResultDTO(AnalysisResult result, OrigImage origImage) {

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy.MM.dd HH:mm");
        String analysisDate = origImage.getAnalysisDate().format(formatter);
        String suitable = (result.isSuitable()) ? "적합" : "부적합";

        return ResultDTO.builder()
                .id(result.getId())
                .origImage(toDataUri(origImage.getImageData()))
                .analysisDate(analysisDate)
                .plastic(result.getPlastic())
                .vinyl(result.getVinyl())
                .wood(result.getWood())
                .total(roundTo2Decimals(result.getWood() + result.getPlastic() + result.getVinyl()))
                .count(result.getCount())
                .suitable(suitable)
                .avgPlastic(roundTo2Decimals(analysisResultRepository.getPlasticAvg()))
                .avgVinyl(roundTo2Decimals(analysisResultRepository.getVinylAvg()))
                .avgWood(roundTo2Decimals(analysisResultRepository.getWoodAvg()))
                .rcnnResult(toDataUri(result.getRcnnResult()))
                .opencvPro(toDataUri(result.getOpencvPro()))
                .opencvResult(toDataUri(result.getOpencvResult()))
                .pca(toDataUri(result.getPca()))
                .build();
    }

    public String toDataUri(byte[] imageBytes) {
        if (imageBytes == null || imageBytes.length == 0) return "";

        String mimeType = "image/jpg";
        try {
            String guessed = URLConnection.guessContentTypeFromStream(new ByteArrayInputStream(imageBytes));
            if (guessed != null) mimeType = guessed;
        } catch (IOException e) {
            // MIME 타입 추정 실패시 기본값 유지
        }
        String base64 = Base64.getEncoder().encodeToString(imageBytes);
        return "data:" + mimeType + ";base64," + base64;
    }

    public double roundTo2Decimals(double value) { return Math.round(value * 100) / 100.0; }

}