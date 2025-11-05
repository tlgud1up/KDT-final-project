package com.example.resource.repository;

import com.example.resource.entity.AnalysisResult;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

public interface AnalysisResultRepository extends JpaRepository<AnalysisResult, Long> {
    @Query("SELECT AVG(a.plastic) FROM AnalysisResult a")
    Double getPlasticAvg();

    @Query("SELECT AVG(a.vinyl) FROM AnalysisResult a")
    Double getVinylAvg();

    @Query("SELECT AVG(a.wood) FROM AnalysisResult a")
    Double getWoodAvg();

    AnalysisResult findByOrigImgId(Long origImgId);

}