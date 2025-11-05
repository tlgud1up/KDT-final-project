package com.example.resource.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AnalysisResponse {
    private int status;
    @JsonProperty("orig_img")
    private String origImg;

    private double plastic;
    private double vinyl;
    private double wood;
    private int count;

    @JsonProperty("rcnn_result")
    private String rcnnResult;
    @JsonProperty("opencv_pro")
    private String opencvPro;
    @JsonProperty("opencv_result")
    private String opencvResult;
    @JsonProperty("pca")
    private String pca;
}