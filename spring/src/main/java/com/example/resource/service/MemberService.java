package com.example.resource.service;

import com.example.resource.dto.SignupRequest;
import com.example.resource.entity.Member;
import com.example.resource.exception.SignupException;
import com.example.resource.repository.MemberRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;


@Service
public class MemberService {

    private final MemberRepository memberRepository;
    private final PasswordEncoder passwordEncoder;

    public MemberService(MemberRepository memberRepository, PasswordEncoder passwordEncoder) {
        this.memberRepository = memberRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @Transactional
    public void register(SignupRequest request) {
        if (memberRepository.findByUsername(request.getUsername()).isPresent()) {
            throw new SignupException("이미 사용 중인 아이디입니다.");
        }

        try {
            Member member = Member.builder()
                    .username(request.getUsername())
                    .password(passwordEncoder.encode(request.getPassword()))
                    .name(request.getName())
                    .birthday(request.getBirthday())
                    .userType(request.getUserType())
                    .build();
            memberRepository.save(member);
        } catch (Exception e) {
            throw new SignupException("회원가입 중 오류가 발생했습니다.");
        }
    }


    public void checkPassword(Long id, Long targetId, String inputPassword) {

        checkIfSameMember(id, targetId);
        Member member = getMemberOrThrow(targetId);

        if (!passwordEncoder.matches(inputPassword, member.getPassword())) {
            throw new IllegalArgumentException("비밀번호가 일치하지 않습니다.");
        }
    }

    @Transactional
    public void updatePW(Long id, Long targetId, String newPassword) {
        checkIfSameMember(id, targetId);
        Member member = getMemberOrThrow(targetId);
        member.setPassword(passwordEncoder.encode(newPassword));

    }

    @Transactional
    public void updateBirthDay(Long id, Long targetId, LocalDate birthday) {
        checkIfSameMember(id, targetId);
        Member member = getMemberOrThrow(targetId);
        member.setBirthday(birthday);
        memberRepository.flush();

    }

    @Transactional
    public void cancelAccount(Long id, Long targetId) {

        checkIfSameMember(id, targetId);
        Member member = getMemberOrThrow(targetId);
        memberRepository.delete(member);
    }

    public void checkIfSameMember(Long id, Long targetId) {
        if (id == null || !id.equals(targetId)) {
            throw new IllegalArgumentException("본인의 계정에만 접근할 수 있습니다.");
        }
    }

    private Member getMemberOrThrow(Long targetId) {
        return memberRepository.findById(targetId)
                .orElseThrow(() -> new IllegalArgumentException("회원 정보를 찾을 수 없습니다."));
    }

}
