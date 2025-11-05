package com.example.resource.security;

import com.example.resource.entity.Member;
import com.example.resource.repository.MemberRepository;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class MemberDetailService implements UserDetailsService {

    private final MemberRepository memberRepository;

    public MemberDetailService(MemberRepository memberRepository) {
        this.memberRepository = memberRepository;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        Optional<Member> optionalUser = memberRepository.findByUsername(username);

        if(optionalUser.isPresent()){
            Member member = optionalUser.get();
            return new MemberDetails(member);
        }

        throw new UsernameNotFoundException(username);
    }

}