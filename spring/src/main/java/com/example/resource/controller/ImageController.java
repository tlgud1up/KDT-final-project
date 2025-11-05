package com.example.resource.controller;

import org.springframework.core.io.InputStreamResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.util.Base64;


@Controller
class ImageController {

    @GetMapping("/upload")
    public String showUploadForm() {
        return "test_upload";
    }

    @GetMapping("/analyze")
    public String analyze() {
        return "newAnalyze2";
    }


//    @PostMapping("/upload")
//    public String handleUpload(@RequestParam("file") MultipartFile file, Model model) throws IOException {
//        MultipartBodyBuilder builder = new MultipartBodyBuilder();
//        builder.part("file", new InputStreamResource(file.getInputStream())).header("Content-Disposition", "form-data; name=file; filename=" + file.getOriginalFilename());
//        WebClient client = WebClient.create("http://localhost:8000");
//        String imageUrl = client.post()
//                .uri("/image/analyze")
//                .contentType(MediaType.MULTIPART_FORM_DATA)
//                .body(BodyInserters.fromMultipartData(builder.build()))
//                .retrieve().bodyToMono(Map.class)
//                .map(m -> (String) m.get("image_url"))
//                .block();
//        model.addAttribute("imageUrl", imageUrl);
//        return "test";
//    }

    @PostMapping("/upload")
    public String handleUpload(@RequestParam("file") MultipartFile file, Model model) throws IOException {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", new InputStreamResource(file.getInputStream()))
                .header("Content-Disposition", "form-data; name=file; filename=" + file.getOriginalFilename());

        WebClient client = WebClient.create("http://192.168.0.63:8000");

        byte[] imageBytes = client.post()
                .uri("/image/analyze")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(builder.build()))
                .retrieve()
                .bodyToMono(byte[].class)
                .doOnError(error -> {
                    System.out.println("에러 발생: " + error.getMessage());
                })
                .block();

        String base64Image = Base64.getEncoder().encodeToString(imageBytes);
        String contentType = file.getContentType(); // 예: image/png
        model.addAttribute("contentType", contentType);
        model.addAttribute("imageData", base64Image);
        return "test";
    }

}
