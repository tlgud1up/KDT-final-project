package com.example.resource.service;

import com.example.resource.dto.RaUsageDTO;
import com.example.resource.entity.RaUsage;
import com.example.resource.repository.RaUsageRepository;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class RaUsageService {

    private final RaUsageRepository raUsageRepository;
    private final MapperService mapperService;
    private final WebClient webClient;

    public RaUsageService(RaUsageRepository raUsageRepository, MapperService mapperService,
                          WebClient raPredictClient) {
        this.raUsageRepository = raUsageRepository;
        this.mapperService = mapperService;
        this.webClient = raPredictClient;
    }

    public List<RaUsageDTO> getDataFromDB(){
        List<RaUsage> data = raUsageRepository.findAll();
        if (data.isEmpty()){
            return getUsageData();
        }
        return mapperService.toRaUsageDTO(data);
    }

    public List<RaUsageDTO> getUsageData() {

        String response = webClient.get()
                .uri("/api/pred")
                .retrieve()
                .bodyToMono(String.class)
                .block();

        List<Map<String, Object>> responseData = parseJsonToList(response);

        List<RaUsageDTO> dtoList = new ArrayList<>();
        for (Map<String, Object> data : responseData) {
            int year = (int) data.get("year");
            double sales = (double) data.get("sales");

            RaUsageDTO dto = RaUsageDTO.builder()
                    .year(year).sales(sales).build();
            dtoList.add(dto);
            RaUsage entity = mapperService.toRaUsage(dto);
            raUsageRepository.save(entity);
        }
        return dtoList;
    }

    private List<Map<String, Object>> parseJsonToList(String response) {
        try {
            // ObjectMapper로 JSON을 List<Map<String, Object>>로 변환
            ObjectMapper objectMapper = new ObjectMapper();
            return objectMapper.readValue(response, new TypeReference<List<Map<String, Object>>>() {
            });
        } catch (Exception e) {
            throw new RuntimeException("제이슨 응답 파싱 에러 :", e);
        }
    }
}