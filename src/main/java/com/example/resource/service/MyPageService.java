package com.example.resource.service;

import com.example.resource.dto.ResultDTO;
import com.example.resource.entity.AnalysisResult;
import com.example.resource.entity.Member;
import com.example.resource.entity.OrigImage;
import com.example.resource.repository.AnalysisResultRepository;
import com.example.resource.repository.OrigImageRepository;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class MyPageService {

    private final MapperService mapperService;
    private final OrigImageRepository origImageRepository;
    private final AnalysisResultRepository analysisResultRepository;

    public MyPageService(MapperService mapperService, OrigImageRepository origImageRepository, AnalysisResultRepository analysisResultRepository) {
        this.mapperService = mapperService;
        this.origImageRepository = origImageRepository;
        this.analysisResultRepository = analysisResultRepository;
    }

    public List<ResultDTO> getResultList(Member member) {
        List<ResultDTO> resultList = new ArrayList<>();

        List<OrigImage> origImages = origImageRepository.findAllByMemberOrderByAnalysisDateDesc(member);

        for (OrigImage origImage : origImages) {
            AnalysisResult result = analysisResultRepository.findByOrigImgId(origImage.getId());
            if (result != null) {
                resultList.add(mapperService.toResultDTO(result, origImage));
            }
        }
        return resultList;
    }
}